

class ImageRetrieval():

    def __init__(self, dim_examples, encoder, train_dataloader_images, device):
        #print("self dim exam", dim_examples)
        self.datastore = faiss.IndexFlatL2(dim_examples) #datastore
        self.encoder= encoder

        #data
        self.device=device
        self.imgs_indexes_of_dataloader = torch.tensor([]).long().to(device)
        #print("self.imgs_indexes_of_dataloader type", self.imgs_indexes_of_dataloader)

        #print("len img dataloader", self.imgs_indexes_of_dataloader.size())
        self._add_examples(train_dataloader_images)
        #print("len img dataloader final", self.imgs_indexes_of_dataloader.size())
        #print("como ficou img dataloader final", self.imgs_indexes_of_dataloader)


    def _add_examples(self, train_dataloader_images):
        print("\nadding input examples to datastore (retrieval)")
        for i, (imgs, imgs_indexes) in enumerate(train_dataloader_images):
            #add to the datastore
            imgs=imgs.to(self.device)
            imgs_indexes = imgs_indexes.long().to(self.device)
            #print("img index type", imgs_indexes)
            encoder_output = self.encoder(imgs)

            encoder_output = encoder_output.view(encoder_output.size()[0], -1, encoder_output.size()[-1])
            input_img = encoder_output.mean(dim=1)
            
            self.datastore.add(input_img.cpu().numpy())

            if i%5==0:
                print("i and img index of ImageRetrival",i, imgs_indexes)
                print("n of examples", self.datastore.ntotal)
            self.imgs_indexes_of_dataloader= torch.cat((self.imgs_indexes_of_dataloader,imgs_indexes))



    def retrieve_nearest_for_train_query(self, query_img, k=2):
        #print("self query img", query_img)
        D, I = self.datastore.search(query_img, k)     # actual search
        #print("all nearest", I)
        #print("I firt", I[:,0])
        #print("if you choose the first", self.imgs_indexes_of_dataloader[I[:,0]])
        nearest_input = self.imgs_indexes_of_dataloader[I[:,1]]
        #print("the nearest input is actual the second for training", nearest_input)
        #nearest_input = I[0,1]
        #print("actual nearest_input", nearest_input)
        return nearest_input

    def retrieve_nearest_for_val_or_test_query(self, query_img, k=1):
        D, I = self.datastore.search(query_img, k)     # actual search
        nearest_input = self.imgs_indexes_of_dataloader[I[:,0]]
        #print("all nearest", I)
        #print("the nearest input", nearest_input)
        return nearest_input